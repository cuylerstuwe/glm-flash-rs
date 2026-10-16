use super::*;
#[test]
fn file_notification_wakes_idle_owner_and_preserves_foreground_priority() -> Result<()> {
    let (broker, owner) = Broker::new();
    let wake = owner.file_wake();
    let (tx, rx) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        tx.send(matches!(owner.next(Duration::from_secs(10)), Action::Tick))
            .unwrap();
        owner
    });
    std::thread::sleep(Duration::from_millis(20));
    wake.notify();
    assert!(rx.recv_timeout(Duration::from_secs(1))?);
    let owner = thread.join().unwrap();
    wake.notify();
    wake.notify();
    let _client = broker.begin(false, Duration::from_secs(10))?;
    assert!(matches!(
        owner.next(Duration::ZERO),
        Action::Prepare(_, Command::Prepare { background: false })
    ));
    // A late completion cannot retain or resurrect a destroyed broker.
    drop(owner);
    drop(_client);
    drop(broker);
    wake.notify();
    Ok(())
}
fn receipt() -> Receipt {
    Receipt {
        applied: 1,
        held: true,
    }
}
fn prepare(o: &Owner) -> Result<()> {
    let a = o.next(Duration::ZERO);
    assert!(matches!(a, Action::Prepare(..)));
    o.applied(a, Some(1), None, receipt(), None)
}
fn submit(o: &Owner) -> Result<()> {
    let a = o.next(Duration::ZERO);
    assert!(matches!(a, Action::Submit(..)));
    o.applied(a, None, Some(99), receipt(), None)
}
#[test]
fn cancelled_image_retains_native_preparation_until_actual_job_exit() -> Result<()> {
    let (broker, owner) = Broker::new();
    let client = broker.begin(false, Duration::from_secs(10))?;
    prepare(&owner)?;
    let image = client.prepare_image()?;
    let encoding = image.job.as_ref().unwrap().encode()?;
    client.cancellation().cancel();
    assert!(image.job.as_ref().unwrap().control.status()?.2);
    assert!(matches!(owner.next(Duration::ZERO), Action::Tick));
    encoding.drained();
    // Downloaded rows remain owned until the preparation itself exits too.
    assert!(matches!(owner.next(Duration::ZERO), Action::Tick));
    drop(image);
    assert!(matches!(owner.next(Duration::ZERO), Action::Release(..)));
    Ok(())
}
#[test]
fn shutdown_waits_for_image_source_and_cancels_its_hold() -> Result<()> {
    let (broker, owner) = Broker::new();
    let client = broker.begin(true, Duration::from_secs(10))?;
    let a = owner.next(Duration::ZERO);
    owner.applied(a, None, None, receipt(), None)?;
    owner.image_hold(true)?;
    let image = client.prepare_image()?;
    assert!(image.job.as_ref().unwrap().control.status()?.0);
    broker.shutdown();
    assert!(image.job.as_ref().unwrap().control.status()?.2);
    assert!(matches!(owner.next(Duration::ZERO), Action::Tick));
    drop(image);
    assert!(matches!(owner.next(Duration::ZERO), Action::Stop));
    Ok(())
}
#[test]
fn grammar_ignore_eos_conflict_never_enters_native_preparation() -> Result<()> {
    let (broker, owner) = Broker::new();
    let client = broker.begin(false, Duration::from_secs(10))?;
    prepare(&owner)?;
    let grammar =
        crate::grammar_bound::Bound::from_request(&serde_json::json!({"guided_regex":"x"}), &[])?
            .unwrap();
    let invalid = crate::output_policy::Policy::new(false, false, &serde_json::Value::Null)?
        .with_ignore_eos(true)?
        .with_grammar(Some(grammar));
    assert!(
        client
            .submit_policy(vec![17].into(), 8, Some(invalid))
            .is_err()
    );
    {
        let state = broker.shared.state.lock().unwrap();
        let entry = &state.entries[&client.id];
        assert!(
            entry.cancel
                && entry.error.is_some()
                && entry.input.is_none()
                && entry.request.is_none()
        );
    }
    assert!(!matches!(owner.next(Duration::ZERO), Action::Submit(..)));
    Ok(())
}
#[test]
fn committed_batches_keep_order_and_reject_invalid_publication_atomically() -> Result<()> {
    let (b, o) = Broker::new();
    let c = b.begin(false, Duration::from_secs(10))?;
    prepare(&o)?;
    c.submit(vec![1].into(), 64)?;
    submit(&o)?;
    for invalid in [vec![1; 17], vec![1, -1], vec![154880]] {
        assert!(o.publish_many(99, &invalid, Some(End::Complete)).is_err());
        let state = b.shared.state.lock().unwrap();
        let e = &state.entries[&c.id];
        assert!(e.output.is_empty() && e.end.is_none() && e.request == Some(99));
    }
    o.publish(99, Some(7), None)?;
    o.publish_many(99, &[8, 9, 10, 11], Some(End::Complete))?;
    for token in 7..=11 {
        assert_eq!(c.receive(Duration::ZERO)?, Some(Event::Token(token)));
    }
    assert_eq!(c.receive(Duration::ZERO)?, Some(Event::End(End::Complete)));
    Ok(())
}
#[test]
fn slow_consumer_never_receives_part_of_a_committed_batch() -> Result<()> {
    let (b, o) = Broker::new();
    let c = b.begin(false, Duration::from_secs(10))?;
    prepare(&o)?;
    c.submit(vec![1].into(), 1024)?;
    submit(&o)?;
    for _ in 0..OUTPUT_BUFFER - 2 {
        o.publish(99, Some(7), None)?;
    }
    o.publish_many(99, &[8, 9, 10], None)?;
    {
        let state = b.shared.state.lock().unwrap();
        let e = &state.entries[&c.id];
        assert!(e.cancel && e.error.is_some());
        assert_eq!(e.output.len(), OUTPUT_BUFFER - 2);
        assert!(e.output.iter().all(|t| *t == 7));
    }
    for _ in 0..OUTPUT_BUFFER - 2 {
        assert_eq!(c.receive(Duration::ZERO)?, Some(Event::Token(7)));
    }
    assert!(c.receive(Duration::ZERO).is_err());
    Ok(())
}
#[test]
fn disconnect_cannot_be_lost_to_full_controls_and_releases_before_cancel() -> Result<()> {
    let (b, o) = Broker::new();
    let c = b.begin(false, Duration::from_secs(10))?;
    prepare(&o)?;
    assert_eq!(c.prepared(Duration::ZERO)?.applied, 1);
    c.submit(vec![1; 64].into(), 4)?;
    submit(&o)?;
    let mut controls = Vec::new();
    for i in 0..CONTROLS {
        controls.push(b.lease("router", i as u64, true, 1.0)?);
    }
    assert!(b.lease("router", 99, true, 1.0).is_err());
    drop(c);
    let a = o.next(Duration::ZERO);
    assert!(matches!(a, Action::Release(_, Command::Release(1))));
    o.applied(a, None, None, receipt(), None)?;
    let a = o.next(Duration::ZERO);
    assert!(matches!(a, Action::Cancel(_, Command::Cancel(99))));
    o.applied(a, None, None, receipt(), None)?;
    o.publish(99, None, Some(End::Cancelled))?;
    assert!(matches!(o.next(Duration::ZERO), Action::Control(..)));
    assert!(b.shared.state.lock().unwrap().entries.is_empty());
    Ok(())
}
#[test]
fn asynchronous_cancel_retains_owner_without_starving_controls_or_worker_ticks() -> Result<()> {
    let (b, o) = Broker::new();
    let c = b.begin(false, Duration::from_secs(10))?;
    prepare(&o)?;
    c.submit(vec![1; 64].into(), 4)?;
    submit(&o)?;
    drop(c);
    let release = o.next(Duration::ZERO);
    assert!(matches!(release, Action::Release(..)));
    o.applied(release, None, None, receipt(), None)?;
    let cancel = o.next(Duration::ZERO);
    assert!(matches!(cancel, Action::Cancel(_, Command::Cancel(99))));
    o.applied(cancel, None, None, receipt(), None)?;
    // Real SSD cancellation cannot retire until the worker has drained and a
    // native hold is released. A repeated Cancel used to starve both forever.
    let control = b.lease("router", 1, false, 1.0)?;
    let action = o.next(Duration::ZERO);
    assert!(matches!(action, Action::Control(..)));
    o.applied(action, None, None, receipt(), None)?;
    assert!(control.try_recv()?.is_ok());
    assert!(matches!(o.next(Duration::ZERO), Action::Tick));
    assert!(matches!(o.next(Duration::ZERO), Action::Tick));
    assert_eq!(b.shared.state.lock().unwrap().entries.len(), 1);
    b.shutdown();
    assert!(matches!(o.next(Duration::ZERO), Action::Tick));
    o.publish(99, None, Some(End::Cancelled))?;
    assert!(matches!(o.next(Duration::ZERO), Action::Stop));
    assert!(b.shared.state.lock().unwrap().entries.is_empty());
    Ok(())
}
#[test]
fn capacity_and_prepare_error_lifetimes() -> Result<()> {
    let (b, o) = Broker::new();
    let mut clients = Vec::new();
    for _ in 0..REQUESTS - 1 {
        clients.push(b.begin(true, Duration::from_secs(10))?);
    }
    assert!(b.begin(true, Duration::from_secs(10)).is_err());
    let fg = b.begin(false, Duration::from_secs(10))?;
    let a = o.next(Duration::ZERO);
    assert!(matches!(
        a,
        Action::Prepare(_, Command::Prepare { background: false })
    ));
    o.applied(a, Some(1), None, receipt(), None)?;
    assert!(fg.submit(vec![1].into(), 0).is_err());
    let a = o.next(Duration::ZERO);
    assert!(matches!(a, Action::Release(..)));
    o.applied(a, None, None, receipt(), None)?;
    drop(fg);
    drop(clients);
    assert!(matches!(o.next(Duration::ZERO), Action::Tick));
    assert!(b.shared.state.lock().unwrap().entries.is_empty());
    Ok(())
}
#[test]
fn slow_output_is_bounded_and_does_not_block_controls() -> Result<()> {
    let (b, o) = Broker::new();
    let c = b.begin(false, Duration::from_secs(10))?;
    prepare(&o)?;
    c.submit(vec![1].into(), 1024)?;
    submit(&o)?;
    let a = o.next(Duration::ZERO);
    o.applied(a, None, None, receipt(), None)?;
    for t in 0..=OUTPUT_BUFFER {
        o.publish(99, Some(t as i32), None)?;
    }
    assert_eq!(
        b.shared.state.lock().unwrap().entries[&c.id].output.len(),
        OUTPUT_BUFFER
    );
    let a = o.next(Duration::ZERO);
    assert!(matches!(a, Action::Cancel(..)));
    o.applied(a, None, None, receipt(), None)?;
    o.publish(99, None, Some(End::Cancelled))?;
    for t in 0..OUTPUT_BUFFER {
        assert_eq!(c.receive(Duration::ZERO)?, Some(Event::Token(t as i32)));
    }
    assert!(c.receive(Duration::ZERO).is_err());
    Ok(())
}
#[test]
fn condvar_wakeup_receipt_order_timeout_and_failstop() -> Result<()> {
    let (b, o) = Broker::new();
    let b2 = b.clone();
    let client = std::thread::spawn(move || -> Result<()> {
        let c = b2.begin(false, Duration::from_secs(10))?;
        let r = c.prepared(Duration::from_secs(2))?;
        ensure!(r.held && r.applied == 1, "invalid native receipt");
        drop(c);
        Ok(())
    });
    let a = o.next(Duration::from_secs(2));
    assert!(matches!(a, Action::Prepare(..)));
    o.applied(a, Some(1), None, receipt(), None)?;
    client.join().unwrap()?;
    let a = o.next(Duration::ZERO);
    assert!(matches!(a, Action::Release(..)));
    o.applied(a, None, None, receipt(), None)?;
    o.next(Duration::ZERO);
    let c = b.begin(false, Duration::from_secs(10))?;
    assert!(c.prepared(Duration::ZERO).is_err());
    o.next(Duration::ZERO);
    assert_eq!(c.receive(Duration::ZERO)?, Some(Event::End(End::Cancelled)));
    let c2 = b.begin(false, Duration::from_millis(2))?;
    prepare(&o)?;
    let a = o.next(Duration::from_secs(1));
    assert!(matches!(a, Action::Release(..)));
    o.applied(a, None, None, receipt(), None)?;
    o.next(Duration::ZERO);
    assert_eq!(c2.receive(Duration::ZERO)?, Some(Event::End(End::TimedOut)));
    let c3 = b.begin(true, Duration::from_secs(10))?;
    let ctl = b.lease("router", 1, true, 1.0)?;
    o.fail("fixture failure");
    assert!(c3.prepared(Duration::ZERO).is_err());
    assert!(ctl.recv()?.is_err());
    assert!(b.begin(false, Duration::from_secs(1)).is_err());
    Ok(())
}
#[test]
fn control_ack_waits_for_application_and_owner_rejects_stale_ids() -> Result<()> {
    let (b, o) = Broker::new();
    let rx = b.lease("r", 1, true, 1.0)?;
    assert!(rx.try_recv().is_err());
    let a = o.next(Duration::ZERO);
    assert!(rx.try_recv().is_err());
    o.applied(a, None, None, receipt(), None)?;
    assert!(rx.recv()?.unwrap().held);
    let c = b.begin(false, Duration::from_secs(10))?;
    let old = c.id;
    drop(c);
    o.next(Duration::ZERO);
    let c = b.begin(false, Duration::from_secs(10))?;
    assert!(c.id > old);
    b.shutdown();
    assert!(matches!(o.next(Duration::ZERO), Action::Stop));
    assert_eq!(c.receive(Duration::ZERO)?, Some(Event::End(End::Cancelled)));
    Ok(())
}

#[test]
fn owner_drop_wakes_waiters_and_shutdown_refuses_unapplied_controls() -> Result<()> {
    let (b, o) = Broker::new();
    let c = b.begin(false, Duration::from_secs(10))?;
    let rx = b.lease("router", 1, true, 1.0)?;
    drop(o);
    assert!(c.prepared(Duration::ZERO).is_err());
    assert!(rx.recv()?.is_err());
    let (b, o) = Broker::new();
    let rx = b.lease("router", 1, true, 1.0)?;
    b.shutdown();
    assert!(rx.recv()?.is_err());
    assert!(matches!(o.next(Duration::ZERO), Action::Stop));
    Ok(())
}

#[test]
fn completed_output_survives_owner_exit() -> Result<()> {
    let (b, o) = Broker::new();
    let c = b.begin(false, Duration::from_secs(10))?;
    prepare(&o)?;
    c.submit(vec![1].into(), 1)?;
    submit(&o)?;
    assert_eq!(c.submitted(Duration::ZERO)?, 99);
    let a = o.next(Duration::ZERO);
    o.applied(a, None, None, receipt(), None)?;
    o.publish(99, Some(7), Some(End::Complete))?;
    drop(o);
    assert_eq!(c.receive(Duration::ZERO)?, Some(Event::Token(7)));
    assert_eq!(c.receive(Duration::ZERO)?, Some(Event::End(End::Complete)));
    Ok(())
}

#[test]
fn submitted_deadline_preserves_native_timeout_reason() -> Result<()> {
    let (b, o) = Broker::new();
    let c = b.begin(false, Duration::from_secs(10))?;
    prepare(&o)?;
    c.submit(vec![1].into(), 4)?;
    submit(&o)?;
    let a = o.next(Duration::ZERO);
    o.applied(a, None, None, receipt(), None)?;
    b.shared
        .state
        .lock()
        .unwrap()
        .entries
        .get_mut(&c.id)
        .unwrap()
        .deadline = Duration::ZERO;
    assert!(matches!(o.next(Duration::ZERO), Action::Tick));
    o.publish(99, None, Some(End::TimedOut))?;
    assert_eq!(c.receive(Duration::ZERO)?, Some(Event::End(End::TimedOut)));
    Ok(())
}

#[test]
fn preparation_expiring_during_native_drain_never_starts() -> Result<()> {
    let (b, o) = Broker::new();
    let c = b.begin(false, Duration::from_secs(10))?;
    let a = o.next(Duration::ZERO);
    assert!(matches!(a, Action::Prepare(..)));
    b.shared
        .state
        .lock()
        .unwrap()
        .entries
        .get_mut(&c.id)
        .unwrap()
        .deadline = Duration::ZERO;
    o.applied(a, Some(1), None, receipt(), None)?;
    assert!(c.prepared(Duration::ZERO).is_err());
    let a = o.next(Duration::ZERO);
    assert!(matches!(a, Action::Release(..)));
    o.applied(a, None, None, receipt(), None)?;
    o.next(Duration::ZERO);
    assert_eq!(c.receive(Duration::ZERO)?, Some(Event::End(End::TimedOut)));
    Ok(())
}

#[test]
fn watchdog_cancellation_is_scoped_and_survives_full_controls() -> Result<()> {
    let (broker, owner) = Broker::new();
    let first = broker.begin(false, Duration::from_secs(10))?;
    let second = broker.begin(false, Duration::from_secs(10))?;
    let cancel = first.cancellation();
    for i in 0..32 {
        let _ = broker.lease(&format!("watch-{i}"), 1, true, 30.0)?;
    }
    cancel.cancel();
    assert!(first.prepared(Duration::ZERO).is_err());
    drop(first);
    let action = owner.next(Duration::ZERO);
    assert!(matches!(action, Action::Prepare(..)));
    owner.applied(
        action,
        Some(1),
        None,
        Receipt {
            applied: 1,
            held: true,
        },
        None,
    )?;
    assert!(second.prepared(Duration::ZERO).is_ok());
    cancel.cancel();
    assert!(second.prepared(Duration::ZERO).is_ok());
    Ok(())
}

#[test]
fn control_rejection_differs_from_owner_failure() -> Result<()> {
    let (b, o) = Broker::new();
    let rx = b.lease("x", 1, true, 30.0)?;
    let action = o.next(Duration::ZERO);
    o.applied(
        action,
        None,
        None,
        receipt(),
        Some("stale lease epoch".into()),
    )?;
    assert!(rx.recv()?.unwrap_err().rejected);
    let rx = b.lease("x", 2, false, 30.0)?;
    o.fail("fixture");
    assert!(!rx.recv()?.unwrap_err().rejected);
    Ok(())
}
#[test]
fn deferred_grammar_keeps_foreground_ticket_until_common_compile_receipt() -> Result<()> {
    let (b, o) = Broker::new();
    let c = b.begin(false, Duration::from_secs(10))?;
    prepare(&o)?;
    let grammar =
        crate::grammar_bound::Bound::from_request(&serde_json::json!({"guided_regex":"x"}), &[])?
            .unwrap();
    let p = crate::output_policy::Policy::new(false, false, &serde_json::Value::Null)?
        .with_grammar(Some(grammar));
    c.submit_policy(vec![1].into(), 8, Some(p))?;
    let action = o.next(Duration::ZERO);
    assert!(
        matches!(action,Action::Submit(id,Command::SubmitGrammar{key,..}) if id==c.id&&key==c.id)
    );
    o.defer_submission(action, c.id)?;
    assert!(matches!(o.next(Duration::ZERO), Action::Tick));
    {
        let state = b.shared.state.lock().unwrap();
        let e = &state.entries[&c.id];
        assert!(
            e.phase == Phase::Compiling
                && e.ticket == Some(1)
                && e.request.is_none()
                && e.assigned.is_none()
                && e.input.is_none()
        );
    }
    o.grammar_applied(c.id, Some(99), None, false)?;
    assert_eq!(c.submitted(Duration::ZERO)?, 99);
    let action = o.next(Duration::ZERO);
    assert!(matches!(action, Action::Release(..)));
    o.applied(action, None, None, receipt(), None)?;
    assert!(o.grammar_applied(c.id, Some(100), None, false).is_err());
    Ok(())
}
#[test]
fn cancelled_grammar_preserves_disconnected_owner_and_ticket_until_retired() -> Result<()> {
    for background in [false, true] {
        let (b, o) = Broker::new();
        let c = b.begin(background, Duration::from_secs(10))?;
        let action = o.next(Duration::ZERO);
        o.applied(
            action,
            if background { None } else { Some(1) },
            None,
            receipt(),
            None,
        )?;
        let grammar = crate::grammar_bound::Bound::from_request(
            &serde_json::json!({"guided_regex":"x"}),
            &[],
        )?
        .unwrap();
        let policy = crate::output_policy::Policy::new(false, false, &serde_json::Value::Null)?
            .with_grammar(Some(grammar));
        c.submit_policy(vec![1].into(), 8, Some(policy))?;
        let key = c.id;
        o.defer_submission(o.next(Duration::ZERO), key)?;
        drop(c);
        b.shutdown();
        for _ in 0..3 {
            let action = o.next(Duration::ZERO);
            assert!(
                matches!(action,Action::CancelGrammar(id,Command::CancelGrammar(k)) if id==key&&k==key)
            );
            o.applied(action, None, None, receipt(), None)?;
            let state = b.shared.state.lock().unwrap();
            let e = &state.entries[&key];
            assert!(e.phase == Phase::Compiling && !e.connected && e.request.is_none());
            assert_eq!(e.ticket, if background { None } else { Some(1) });
        }
        o.grammar_applied(
            key,
            None,
            Some("cancelled after actual compiler retirement".into()),
            false,
        )?;
        if !background {
            let action = o.next(Duration::ZERO);
            assert!(matches!(action, Action::Release(_, Command::Release(1))));
            o.applied(action, None, None, receipt(), None)?;
        }
        assert!(matches!(o.next(Duration::ZERO), Action::Stop));
        assert!(b.shared.state.lock().unwrap().entries.is_empty());
    }
    Ok(())
}
#[test]
fn grammar_failure_has_no_native_request_identity_and_does_not_break_next_client() -> Result<()> {
    let (b, o) = Broker::new();
    let c = b.begin(false, Duration::from_secs(10))?;
    prepare(&o)?;
    let grammar =
        crate::grammar_bound::Bound::from_request(&serde_json::json!({"guided_regex":"["}), &[])?
            .unwrap();
    let policy = crate::output_policy::Policy::new(false, false, &serde_json::Value::Null)?
        .with_grammar(Some(grammar));
    c.submit_policy(vec![1].into(), 8, Some(policy))?;
    o.defer_submission(o.next(Duration::ZERO), c.id)?;
    assert!(o.grammar_applied(c.id, None, None, false).is_err());
    assert!(
        o.grammar_applied(c.id, Some(9), Some("both".into()), false)
            .is_err()
    );
    o.grammar_applied(c.id, None, Some("invalid grammar".into()), true)?;
    assert!(
        c.submitted(Duration::ZERO)
            .unwrap_err()
            .downcast_ref::<InvalidGrammar>()
            .is_some()
    );
    o.applied(o.next(Duration::ZERO), None, None, receipt(), None)?;
    drop(c);
    let d = b.begin(false, Duration::from_secs(10))?;
    prepare(&o)?;
    d.submit(vec![1].into(), 8)?;
    submit(&o)?;
    assert_eq!(d.submitted(Duration::ZERO)?, 99);
    Ok(())
}
